import type { SVGProps } from 'react';

export function ML(props: SVGProps<SVGSVGElement>) {
    return (
        <svg
            viewBox="0 0 32 32"
            height="128"
            width="128"
            xmlns="http://www.w3.org/2000/svg"
            {...props}
        >
            <rect width="32" height="32" rx="6" ry="6" fill="#000000" />
            <text
                x="16"
                y="16"
                fill="#ffffff"
                fontFamily="-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
                fontSize="14"
                fontWeight="700"
                textAnchor="middle"
                dominantBaseline="central"
                letterSpacing="-0.5"
            >
                ML
            </text>
        </svg>
    );
}
